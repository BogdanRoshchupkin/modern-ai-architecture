from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator

import ftfy
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 42

SPACE_RE = re.compile(r"[ \t\r\f\v]+")
NEWLINE_RE = re.compile(r"\n{3,}")
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
RESIDUAL_HTML_RE = re.compile(r"<[^>\n]{1,500}>")
RESIDUAL_TAG_START_RE = re.compile(r"<\s*/?\s*[a-zA-Z][^>\n]{0,500}")
BBCODE_RE = re.compile(r"\[(?:/?(?:code|php|html|url|email|quote|indent|video)|[a-z]+=[^\]]+)\]", re.I)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
DEFAULT_TOXIC_KEYWORDS = {
    "kill yourself",
    "racial slur",
}


def normalize_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = RESIDUAL_HTML_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    text = "\n".join(part.strip() for part in text.splitlines())
    text = NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def is_allowed_language(text: str, allowlist: set[str]) -> bool:
    if not allowlist:
        return True
    try:
        return detect(text) in allowlist
    except LangDetectException:
        return False


def contains_blocked_keyword(text: str, keywords: set[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def is_markup_or_boilerplate(text: str) -> bool:
    lowered = text.lower()
    if RESIDUAL_TAG_START_RE.search(text) or len(BBCODE_RE.findall(text)) >= 3:
        return True
    cookie_mentions = lowered.count("cookie")
    ad_boilerplate = "advertis" in lowered or "user data" in lowered or "tracking" in lowered
    return cookie_mentions >= 3 and ad_boilerplate


def has_too_many_foreign_symbols(text: str, allowlist: set[str], max_ratio: float = 0.02) -> bool:
    if "en" not in allowlist:
        return False
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return True
    cjk_count = len(CJK_RE.findall(text))
    return cjk_count / len(letters) > max_ratio


def split_long_text(text: str, max_words: int) -> list[str]:
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def clean_records(
    records: Iterable[dict],
    language_allowlist: Iterable[str] = ("en",),
    min_words: int = 20,
    max_words: int = 650,
    toxic_keywords: Iterable[str] | None = None,
) -> Iterator[dict]:
    allowlist = set(language_allowlist)
    blocked = set(toxic_keywords or [])
    for row in records:
        text = normalize_text(row.get("text", ""))
        if not text or contains_blocked_keyword(text, blocked):
            continue
        if (
            word_count(text) < min_words
            or is_markup_or_boilerplate(text)
            or has_too_many_foreign_symbols(text, allowlist)
            or not is_allowed_language(text[:4000], allowlist)
        ):
            continue
        for index, chunk in enumerate(split_long_text(text, max_words)):
            chunk = normalize_text(chunk)
            if (
                word_count(chunk) >= min_words
                and not is_markup_or_boilerplate(chunk)
                and not has_too_many_foreign_symbols(chunk, allowlist)
                and is_allowed_language(chunk[:4000], allowlist)
            ):
                result = dict(row)
                result["text"] = chunk
                result["chunk_id"] = index
                yield result
