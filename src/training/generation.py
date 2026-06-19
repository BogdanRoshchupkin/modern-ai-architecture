from __future__ import annotations

import torch
from omegaconf import DictConfig

from src.tokenization.bpe import BpeTokenizer
from src.training.lightning_module import GPTLightningModule


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
