from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from src.utils.io import ensure_parent


class BpeTokenizer:
    def __init__(self, vocab_size: int = 8000, special_tokens: list[str] | None = None) -> None:
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>"]
        self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.vocab_size_target = vocab_size

    def train(self, texts: list[str]) -> None:
        trainer = BpeTrainer(vocab_size=self.vocab_size_target, special_tokens=self.special_tokens)
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def save(self, path: str | Path) -> Path:
        target = ensure_parent(path)
        self.tokenizer.save(str(target))
        return target

    @classmethod
    def load(cls, path: str | Path) -> "BpeTokenizer":
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(str(path))
        return instance

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
