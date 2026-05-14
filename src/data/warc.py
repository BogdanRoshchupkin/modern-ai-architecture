from __future__ import annotations

from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup
from warcio.archiveiterator import ArchiveIterator


def html_to_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    return soup.get_text("\n")


def iter_warc_texts(path: str | Path, limit: int | None = None) -> Iterator[dict]:
    emitted = 0
    with Path(path).open("rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != "response":
                continue

            payload = record.content_stream().read()
            if not payload:
                continue

            content_type = record.http_headers.get_header("Content-Type") if record.http_headers else ""
            if "html" not in (content_type or "").lower():
                continue

            text = html_to_text(payload)
            url = record.rec_headers.get_header("WARC-Target-URI")
            yield {"id": record.rec_headers.get_header("WARC-Record-ID"), "url": url, "text": text}
            emitted += 1
            if limit is not None and emitted >= limit:
                return
