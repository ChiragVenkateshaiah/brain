"""Tests for brain.chunker."""

import math

import pytest

from brain.chunker import chunk_text


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
