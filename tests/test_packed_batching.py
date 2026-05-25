from src.data.packed_batching import pack_sequences
import torch


def test_pack_sequences_combines_short_objects_and_masks_segments():
    input_ids, attention_mask = pack_sequences([[1, 2, 3], [4, 5]], max_length=6, pad_id=0)

    assert torch.equal(input_ids, torch.tensor([[1, 2, 3, 4, 5, 0]]))
    assert torch.equal(attention_mask, torch.tensor([[1, 1, 1, 2, 2, 0]]))


def test_pack_sequences_splits_long_objects():
    input_ids, attention_mask = pack_sequences([[1, 2, 3, 4, 5]], max_length=3, pad_id=0)

    assert torch.equal(input_ids, torch.tensor([[1, 2, 3], [4, 5, 0]]))
    assert torch.equal(attention_mask, torch.tensor([[1, 1, 1], [1, 1, 0]]))
