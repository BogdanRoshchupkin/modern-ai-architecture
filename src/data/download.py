from __future__ import annotations

import gzip
import io
from pathlib import Path

import requests
from tqdm import tqdm

from src.utils.io import ensure_parent


def resolve_common_crawl_warc_url(
    snapshot: str,
    base_url: str = "https://data.commoncrawl.org/",
    warc_index: int = 0,
) -> str:
    """Resolve a real WARC URL from Common Crawl's official path manifest."""
    base_url = base_url.rstrip("/") + "/"
    manifest_url = f"{base_url}crawl-data/{snapshot}/warc.paths.gz"
    with requests.get(manifest_url, timeout=60) as response:
        response.raise_for_status()
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as archive:
            paths = [line.decode("utf-8").strip() for line in archive if line.strip()]

    if not paths:
        raise ValueError(f"No WARC paths found in {manifest_url}")
    if warc_index < 0 or warc_index >= len(paths):
        raise IndexError(f"WARC index {warc_index} is outside manifest size {len(paths)}")
    return base_url + paths[warc_index]


def download_file(url: str, output_path: str | Path, chunk_size: int = 1024 * 1024) -> Path:
    target = ensure_parent(output_path)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with target.open("wb") as sink, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"download {target.name}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    sink.write(chunk)
                    progress.update(len(chunk))
    return target
