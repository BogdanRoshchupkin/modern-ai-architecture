from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


class CharTokenizer:
    def __init__(self) -> None:
        self.stoi: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.itos: list[str] = ["<PAD>", "<UNK>"]

    def train(self, texts: Iterable[str]) -> None:
        chars = sorted(set("".join(texts)))
        self.itos = ["<PAD>", "<UNK>"] + chars
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(char, self.stoi["<UNK>"]) for char in text]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


class WordTokenizer:
    def __init__(self, max_vocab_size: int | None = None, min_frequency: int = 1) -> None:
        self.max_vocab_size = max_vocab_size
        self.min_frequency = min_frequency
        self.stoi: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.itos: list[str] = ["<PAD>", "<UNK>"]

    def train(self, texts: Iterable[str]) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(TOKEN_RE.findall(text.lower()))
        tokens = [token for token, count in counter.most_common() if count >= self.min_frequency]
        if self.max_vocab_size is not None:
            tokens = tokens[: max(0, self.max_vocab_size - 2)]
        self.itos = ["<PAD>", "<UNK>"] + tokens
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(token.lower(), self.stoi["<UNK>"]) for token in TOKEN_RE.findall(text)]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)
