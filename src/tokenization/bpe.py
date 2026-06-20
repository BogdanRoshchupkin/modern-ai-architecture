from __future__ import annotations

import json
import re
from hashlib import sha256
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from src.utils.io import ensure_parent

TOKEN_RE = re.compile(r"\S+", flags=re.UNICODE)
END_OF_WORD = "</w>"


def bpe_file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


class BpeTokenizer:
    """Small from-scratch byte-pair tokenizer for the lab.

    It learns merges over whitespace-delimited words represented as character
    tuples with an explicit end-of-word marker. This is intentionally compact,
    but the core BPE algorithm is implemented here rather than delegated to a
    ready-made tokenizer library.
    """

    def __init__(self, vocab_size: int = 8000, special_tokens: list[str] | None = None) -> None:
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>"]
        self.vocab_size_target = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}
        self.encode_cache: dict[str, list[int]] = {}
        self.stoi: dict[str, int] = {token: index for index, token in enumerate(self.special_tokens)}
        self.itos: list[str] = list(self.special_tokens)
        self.unk_token = "<UNK>"

    def train(
        self,
        texts: Iterable[str],
        max_documents: int | None = None,
        min_pair_frequency: int = 2,
    ) -> None:
        word_frequencies: Counter[tuple[str, ...]] = Counter()
        for document_index, text in enumerate(texts):
            if max_documents is not None and document_index >= max_documents:
                break
            for token in TOKEN_RE.findall(text):
                word_frequencies[tuple(token) + (END_OF_WORD,)] += 1

        self.merges = []
        vocab = set(self.special_tokens)
        for word in word_frequencies:
            vocab.update(word)

        while len(vocab) < self.vocab_size_target:
            pair_counts = self._get_pair_counts(word_frequencies)
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_pair_frequency:
                break
            merged_token = "".join(best_pair)
            word_frequencies = self._merge_pair(word_frequencies, best_pair, merged_token)
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1
            vocab.add(merged_token)

        ordered_tokens = list(self.special_tokens)
        learned_tokens = sorted(vocab - set(self.special_tokens))
        ordered_tokens.extend(learned_tokens[: max(0, self.vocab_size_target - len(ordered_tokens))])
        self.itos = ordered_tokens
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @staticmethod
    def _get_pair_counts(words: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
        counts: Counter[tuple[str, str]] = Counter()
        for word, frequency in words.items():
            for index in range(len(word) - 1):
                counts[(word[index], word[index + 1])] += frequency
        return counts

    @staticmethod
    def _merge_pair(
        words: Counter[tuple[str, ...]],
        pair: tuple[str, str],
        merged_token: str,
    ) -> Counter[tuple[str, ...]]:
        result: Counter[tuple[str, ...]] = Counter()
        left, right = pair
        for word, frequency in words.items():
            merged_word: list[str] = []
            index = 0
            while index < len(word):
                if index < len(word) - 1 and word[index] == left and word[index + 1] == right:
                    merged_word.append(merged_token)
                    index += 2
                else:
                    merged_word.append(word[index])
                    index += 1
            result[tuple(merged_word)] += frequency
        return result

    def _encode_word(self, token: str) -> list[str]:
        pieces = tuple(token) + (END_OF_WORD,)
        while len(pieces) > 1:
            candidate_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(pieces, pieces[1:])
                if pair in self.merge_ranks
            ]
            if not candidate_pairs:
                break
            _, (left, right) = min(candidate_pairs, key=lambda item: item[0])
            merged = "".join((left, right))
            merged_pieces: list[str] = []
            index = 0
            while index < len(pieces):
                if index < len(pieces) - 1 and pieces[index] == left and pieces[index + 1] == right:
                    merged_pieces.append(merged)
                    index += 2
                else:
                    merged_pieces.append(pieces[index])
                    index += 1
            pieces = tuple(merged_pieces)
        return list(pieces)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in TOKEN_RE.findall(text):
            if token not in self.encode_cache:
                unk_id = self.stoi[self.unk_token]
                self.encode_cache[token] = [self.stoi.get(piece, unk_id) for piece in self._encode_word(token)]
            ids.extend(self.encode_cache[token])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        pieces = [self.itos[token_id] for token_id in ids if 0 <= token_id < len(self.itos)]
        text = "".join(piece for piece in pieces if piece not in self.special_tokens)
        return text.replace(END_OF_WORD, " ").strip()

    def save(self, path: str | Path) -> Path:
        target = ensure_parent(path)
        payload = {
            "type": "custom_bpe",
            "vocab_size_target": self.vocab_size_target,
            "special_tokens": self.special_tokens,
            "merges": self.merges,
            "vocab": self.stoi,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "BpeTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "custom_bpe":
            raise ValueError(
                "Unsupported BPE file format. Re-train the tokenizer with the custom BpeTokenizer."
            )
        instance = cls(
            vocab_size=int(payload["vocab_size_target"]),
            special_tokens=list(payload["special_tokens"]),
        )
        instance.merges = [tuple(pair) for pair in payload["merges"]]
        instance.merge_ranks = {pair: index for index, pair in enumerate(instance.merges)}
        instance.encode_cache = {}
        instance.stoi = {str(token): int(index) for token, index in payload["vocab"].items()}
        instance.itos = [""] * len(instance.stoi)
        for token, index in instance.stoi.items():
            instance.itos[index] = token
        return instance

    @property
    def vocab_size(self) -> int:
        return len(self.itos)
