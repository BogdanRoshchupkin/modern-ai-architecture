import torch

from src.models.gpt import (
    GPTConfig,
    GPTLikeModel,
    block_causal_attention_mask,
    language_model_loss_mask,
    masked_language_model_loss,
    packed_position_ids,
)


def test_packed_position_ids_reset_for_each_segment():
    segment_ids = torch.tensor([[1, 1, 2, 2, 2, 0]])

    positions = packed_position_ids(segment_ids)

    assert positions.tolist() == [[0, 1, 0, 1, 2, 0]]


def test_block_causal_attention_mask_prevents_cross_segment_attention():
    segment_ids = torch.tensor([[1, 1, 2, 2, 0]])

    mask = block_causal_attention_mask(segment_ids)[0].int().tolist()

    assert mask == [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]


def test_language_model_loss_mask_skips_segment_boundaries_and_padding():
    segment_ids = torch.tensor([[1, 1, 2, 2, 0]])

    loss_mask = language_model_loss_mask(segment_ids)

    assert loss_mask.tolist() == [[True, False, True, False]]


def test_gpt_forward_and_masked_loss_are_finite():
    config = GPTConfig(vocab_size=16, max_seq_len=8, d_model=16, n_layers=1, n_heads=4, d_ff=32, dropout=0.0)
    model = GPTLikeModel(config)
    input_ids = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]])
    segment_ids = torch.tensor([[1, 1, 2, 2, 0, 0, 0, 0]])

    logits = model(input_ids, segment_ids)
    loss = masked_language_model_loss(logits, input_ids, segment_ids)

    assert logits.shape == (1, 8, 16)
    assert torch.isfinite(loss)
    assert loss < 10
