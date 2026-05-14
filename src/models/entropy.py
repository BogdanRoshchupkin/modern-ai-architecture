from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class Gpt2EntropyScorer:
    def __init__(self, model_name: str = "gpt2", device: str | None = None, max_length: int = 1024):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, text: str) -> dict:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
            return {"entropy": float("nan"), "token_count": int(input_ids.shape[1])}
        output = self.model(input_ids=input_ids, labels=input_ids)
        return {"entropy": float(output.loss.detach().cpu()), "token_count": int(input_ids.shape[1])}


def add_entropy_scores(records: Iterable[dict], scorer: Gpt2EntropyScorer) -> Iterator[dict]:
    for row in tqdm(records, desc="score entropy"):
        scored = dict(row)
        scored.update(scorer.score(row["text"]))
        yield scored


def dataset_information_density(records: list[dict]) -> float:
    weights = np.array([row.get("token_count", 0) for row in records], dtype=np.float64)
    values = np.array([row.get("entropy", np.nan) for row in records], dtype=np.float64)
    valid = np.isfinite(values) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def filter_by_entropy_quantiles(records: list[dict], low: float = 0.02, high: float = 0.98) -> list[dict]:
    values = np.array([row["entropy"] for row in records if np.isfinite(row.get("entropy", np.nan))])
    if len(values) == 0:
        return []
    low_value, high_value = np.quantile(values, [low, high])
    return [row for row in records if low_value <= row.get("entropy", float("inf")) <= high_value]


def deduplicate(records: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for row in records:
        text = row["text"].strip()
        key = " ".join(text.lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique
