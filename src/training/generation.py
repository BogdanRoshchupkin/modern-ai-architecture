from __future__ import annotations

import warnings

import torch
from omegaconf import DictConfig

from src.tokenization.bpe import BpeTokenizer, bpe_file_sha256
from src.training.lightning_module import GPTLightningModule


def validate_checkpoint_tokenizer(checkpoint_path: str, tokenizer_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    paths = checkpoint.get("hyper_parameters", {}).get("paths", {})
    expected_hash = paths.get("tokenizer_sha256")
    if not expected_hash:
        warnings.warn(
            "Checkpoint does not contain tokenizer_sha256. "
            "Make sure the tokenizer file is from the same training run.",
            stacklevel=2,
        )
        return
    actual_hash = bpe_file_sha256(tokenizer_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "Tokenizer fingerprint does not match the checkpoint. "
            "Use the common_crawl_bpe.json saved with this checkpoint or retrain the model. "
            f"checkpoint tokenizer_sha256={expected_hash}, current tokenizer_sha256={actual_hash}"
        )


@torch.no_grad()
def generate_text(
    checkpoint_path: str,
    config: DictConfig,
    prompt: str,
    tokenizer_path: str,
    max_new_tokens: int = 80,
    temperature: float = 0.9,
    top_k: int = 50,
    device: str | None = None,
    use_kv_cache: bool = True,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    validate_checkpoint_tokenizer(checkpoint_path, tokenizer_path)
    tokenizer = BpeTokenizer.load(tokenizer_path)
    model = GPTLightningModule.load_from_checkpoint(checkpoint_path, config=config).to(device)
    model.eval()

    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [int(config.model.pad_token_id)]
    max_seq_len = int(config.model.max_seq_len)
    input_ids = torch.tensor(ids[-max_seq_len:], dtype=torch.long, device=device).unsqueeze(0)
    segment_ids = torch.ones_like(input_ids)
    generated_ids = input_ids.squeeze(0).tolist()

    if use_kv_cache:
        logits, past_key_values = model(input_ids, segment_ids, use_cache=True)
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                values, indices = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                filtered = torch.full_like(next_logits, -float("inf"))
                next_logits = filtered.scatter(1, indices, values)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated_ids.append(int(next_id.item()))
            logits, past_key_values = model(
                next_id,
                torch.ones_like(next_id),
                past_key_values=past_key_values,
                use_cache=True,
            )
        return tokenizer.decode(generated_ids)

    for _ in range(max_new_tokens):
        input_ids = input_ids[:, -max_seq_len:]
        segment_ids = torch.ones_like(input_ids)
        logits = model(input_ids, segment_ids)[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
            filtered = torch.full_like(logits, -float("inf"))
            logits = filtered.scatter(1, indices, values)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(input_ids.squeeze(0).tolist())
