"""Tests for brain.parsers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from brain.parsers import ParseError, parse_html, parse_pdf


# --- PDF ---


def test_parse_pdf_extracts_text(tmp_path):
    fake_page = MagicMock()
    fake_page.extract_text.return_value = "hello world"
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    with patch("brain.parsers.pypdf.PdfReader", return_value=fake_reader):
        result = parse_pdf(tmp_path / "doc.pdf")

    assert "hello world" in result


def test_parse_pdf_multipage_joined_with_double_newline(tmp_path):
    pages = [MagicMock(), MagicMock()]
    pages[0].extract_text.return_value = "page one"
    pages[1].extract_text.return_value = "page two"
    fake_reader = MagicMock()
    fake_reader.pages = pages

    with patch("brain.parsers.pypdf.PdfReader", return_value=fake_reader):
        result = parse_pdf(tmp_path / "doc.pdf")

    assert result == "page one\n\npage two"


def test_parse_pdf_none_extract_text_treated_as_empty(tmp_path):
    fake_page = MagicMock()
    fake_page.extract_text.return_value = None
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    with patch("brain.parsers.pypdf.PdfReader", return_value=fake_reader):
        result = parse_pdf(tmp_path / "doc.pdf")

    assert result == ""


def test_parse_pdf_raises_parse_error_on_corrupt(tmp_path):
    from pypdf.errors import PdfReadError

    with patch("brain.parsers.pypdf.PdfReader", side_effect=PdfReadError("bad")):
        with pytest.raises(ParseError):
            parse_pdf(tmp_path / "bad.pdf")


# --- HTML ---


def test_parse_html_strips_tags(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<html><body><p>hello world</p></body></html>")
    assert "hello world" in parse_html(f)
    assert "<p>" not in parse_html(f)


def test_parse_html_removes_script_content(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<html><body><script>evil()</script><p>visible</p></body></html>")
    result = parse_html(f)
    assert "evil()" not in result
    assert "visible" in result


def test_parse_html_removes_style_content(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("<html><head><style>body{color:red}</style></head><body>text</body></html>")
    result = parse_html(f)
    assert "color" not in result
    assert "text" in result


def test_parse_html_handles_malformed(tmp_path):
    f = tmp_path / "broken.html"
    f.write_text("<p>unclosed <b>tag <div>content")
    result = parse_html(f)
    assert "content" in result
