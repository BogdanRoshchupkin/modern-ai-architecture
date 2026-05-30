from __future__ import annotations

from typing import Iterable

import torch


def pack_sequences(
    sequences: Iterable[list[int]],
    max_length: int = 512,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed_rows: list[list[int]] = []
    masks: list[list[int]] = []
    current: list[int] = []
    current_mask: list[int] = []
    segment_id = 1

    def flush() -> None:
        nonlocal current, current_mask, segment_id
        if not current:
            return
        pad_size = max_length - len(current)
        packed_rows.append(current + [pad_id] * pad_size)
        masks.append(current_mask + [0] * pad_size)
        current = []
        current_mask = []
        segment_id = 1

    for sequence in sequences:
        start = 0
        while start < len(sequence):
            remaining = max_length - len(current)
            if remaining == 0:
                flush()
                remaining = max_length
            chunk = sequence[start : start + remaining]
            current.extend(chunk)
            current_mask.extend([segment_id] * len(chunk))
            start += len(chunk)
            if start < len(sequence) or len(current) == max_length:
                flush()
            else:
                segment_id += 1
    flush()
    return torch.tensor(packed_rows, dtype=torch.long), torch.tensor(masks, dtype=torch.long)
