"""Format parsers: PDF and HTML → plain text."""

from pathlib import Path

import pypdf
from pypdf.errors import PdfReadError
from bs4 import BeautifulSoup


class ParseError(Exception):
    pass


def parse_pdf(path: Path) -> str:
    try:
        reader = pypdf.PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except PdfReadError as e:
        raise ParseError(f"failed to parse PDF {path}: {e}") from e


def parse_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)
