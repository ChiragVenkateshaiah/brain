"""Tests for brain.chunker."""

import math

import pytest

from brain.chunker import chunk_markdown, chunk_text


def test_empty_string_returns_empty():
    assert chunk_text("") == []


def test_whitespace_only_returns_empty():
    assert chunk_text("   \n\t  ") == []


def test_short_text_single_chunk():
    assert chunk_text("hello") == ["hello"]


def test_text_exactly_target_size_single_chunk():
    text = "a" * 500
    result = chunk_text(text, target_size=500)
    assert result == [text]


def test_long_text_produces_multiple_chunks():
    text = "x" * 1000
    result = chunk_text(text, target_size=500, overlap=50)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk) <= 500


def test_chunk_count_matches_expected():
    text = "a" * 1000
    target, overlap = 500, 50
    stride = target - overlap
    # ceil((len - overlap) / stride) simplifies to how many strides cover
    expected = math.ceil((len(text) - overlap) / stride)
    result = chunk_text(text, target_size=target, overlap=overlap)
    assert len(result) == expected


def test_overlap_bytes_match_between_consecutive_chunks():
    text = "abcdefghij" * 100  # 1000 chars
    result = chunk_text(text, target_size=100, overlap=20)
    for i in range(len(result) - 1):
        # end of chunk i should equal start of chunk i+1
        assert result[i][-20:] == result[i + 1][:20]


def test_last_chunk_may_be_shorter():
    text = "a" * 550
    result = chunk_text(text, target_size=500, overlap=0)
    assert len(result) == 2
    assert len(result[1]) == 50


def test_no_overlap_no_repeated_chars():
    text = "abcdefghij"
    result = chunk_text(text, target_size=3, overlap=0)
    assert "".join(result) == text


def test_invalid_target_size_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", target_size=0)


def test_negative_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", target_size=10, overlap=-1)


def test_overlap_gte_target_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", target_size=10, overlap=10)


# --- chunk_markdown ---


def test_chunk_markdown_empty_returns_empty():
    assert chunk_markdown("") == []


def test_chunk_markdown_no_headers_matches_chunk_text():
    text = "word " * 300
    assert chunk_markdown(text) == chunk_text(text)


def test_chunk_markdown_single_section_short_body():
    result = chunk_markdown("# Heading\n\nshort body")
    assert len(result) == 1
    assert result[0].startswith("# Heading")
    assert "short body" in result[0]


def test_chunk_markdown_heading_only_emits_chunk():
    result = chunk_markdown("# Heading\n\n")
    assert len(result) == 1
    assert result[0] == "# Heading"


def test_chunk_markdown_multiple_sections_each_has_own_heading():
    text = "# Alpha\n\nalpha content\n\n# Beta\n\nbeta content"
    result = chunk_markdown(text)
    alpha_chunks = [c for c in result if "alpha content" in c]
    beta_chunks = [c for c in result if "beta content" in c]
    assert alpha_chunks and all("# Alpha" in c for c in alpha_chunks)
    assert beta_chunks and all("# Beta" in c for c in beta_chunks)


def test_chunk_markdown_long_section_splits_each_chunk_has_heading():
    text = "# Section\n\n" + "x " * 800
    result = chunk_markdown(text, target_size=200, overlap=20)
    assert len(result) > 1
    assert all(c.startswith("# Section") for c in result)


def test_chunk_markdown_preamble_before_first_heading_kept():
    text = "intro text\n\n# Heading\n\nbody"
    result = chunk_markdown(text)
    assert any("intro text" in c for c in result)
    assert any("body" in c for c in result)


def test_chunk_markdown_invalid_target_size_raises():
    with pytest.raises(ValueError):
        chunk_markdown("# H\n\nbody", target_size=0)


def test_chunk_markdown_overlap_gte_target_raises():
    with pytest.raises(ValueError):
        chunk_markdown("# H\n\nbody", target_size=10, overlap=10)
