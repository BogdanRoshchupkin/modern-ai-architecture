import torch
from omegaconf import OmegaConf

from src.models.gpt import GPTLikeModel


def _gqa_config():
    return OmegaConf.create(
        {
            "vocab_size": 32,
            "max_seq_len": 16,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "pad_token_id": 0,
        }
    )


def test_gqa_cache_uses_grouped_key_value_heads():
    model = GPTLikeModel(_gqa_config()).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    segment_ids = torch.ones_like(input_ids)

    logits, cache = model(input_ids, segment_ids, use_cache=True)

    assert logits.shape == (1, 4, 32)
    assert len(cache) == 2
    key_cache, value_cache = cache[0]
    assert key_cache.shape == (1, 2, 4, 8)
    assert value_cache.shape == (1, 2, 4, 8)


def test_kv_cache_incremental_logits_match_full_forward():
    torch.manual_seed(0)
    model = GPTLikeModel(_gqa_config()).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    segment_ids = torch.ones_like(input_ids)

    with torch.no_grad():
        full_logits = model(input_ids, segment_ids)
        _, cache = model(input_ids[:, :-1], segment_ids[:, :-1], use_cache=True)
        cached_logits, _ = model(
            input_ids[:, -1:],
            segment_ids[:, -1:],
            past_key_values=cache,
            use_cache=True,
        )

    torch.testing.assert_close(cached_logits[:, -1, :], full_logits[:, -1, :], atol=1e-5, rtol=1e-5)
