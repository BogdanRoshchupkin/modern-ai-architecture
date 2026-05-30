from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def _config_value(config: Any, key: str, default: Any | None = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def packed_position_ids(segment_ids: torch.Tensor) -> torch.Tensor:
    """Return per-segment positions reset to zero for each packed object."""
    token_indices = torch.arange(segment_ids.size(1), device=segment_ids.device).expand_as(segment_ids)
    segment_starts = torch.ones_like(segment_ids, dtype=torch.bool)
    segment_starts[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    segment_starts |= segment_ids == 0
    start_indices = torch.where(segment_starts, token_indices, torch.zeros_like(token_indices))
    active_starts = torch.cummax(start_indices, dim=1).values
    positions = token_indices - active_starts
    return positions.masked_fill(segment_ids == 0, 0)


def block_causal_attention_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Build M[i,j] = same_segment(i,j) and j <= i and segment != 0."""
    seq_len = segment_ids.size(1)
    same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
    non_pad = segment_ids.unsqueeze(2) != 0
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=segment_ids.device).tril()
    return same_segment & non_pad & causal.unsqueeze(0)


def language_model_loss_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    current_segments = segment_ids[:, :-1]
    next_segments = segment_ids[:, 1:]
    return (current_segments == next_segments) & (current_segments != 0)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encodings = torch.zeros(max_seq_len, d_model)
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encodings", encodings, persistent=False)

    def forward(self, token_embeddings: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        position_ids = packed_position_ids(segment_ids).clamp_max(self.encodings.size(0) - 1)
        positional = self.encodings[position_ids]
        return self.dropout(token_embeddings + positional)


class MultiHeadMaskedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = block_causal_attention_mask(segment_ids).unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.resid_dropout(self.out_proj(output))
        return output * (segment_ids != 0).unsqueeze(-1)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Post-norm transformer block required by the lab statement."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        d_model = int(_config_value(config, "d_model"))
        n_heads = int(_config_value(config, "n_heads"))
        d_ff = int(_config_value(config, "d_ff"))
        dropout = float(_config_value(config, "dropout"))
        self.attention = MultiHeadMaskedSelfAttention(d_model, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        z = self.attn_norm(x + self.attention(x, segment_ids))
        return self.ffn_norm(z + self.ffn(z))


class GPTLikeModel(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        vocab_size = int(_config_value(config, "vocab_size"))
        max_seq_len = int(_config_value(config, "max_seq_len", 512))
        d_model = int(_config_value(config, "d_model", 256))
        n_layers = int(_config_value(config, "n_layers", 4))
        dropout = float(_config_value(config, "dropout", 0.1))
        pad_token_id = int(_config_value(config, "pad_token_id", 0))
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_encoding = SinusoidalPositionalEncoding(
            d_model,
            max_seq_len,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        x = self.position_encoding(x, segment_ids)
        for block in self.blocks:
            x = block(x, segment_ids)
        return self.lm_head(self.final_norm(x))


class MaskedLanguageModelingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward_with_count(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction_logits = logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        loss_mask = language_model_loss_mask(segment_ids)
        losses = self.cross_entropy(
            prediction_logits.view(-1, prediction_logits.size(-1)),
            labels.view(-1),
        ).view_as(labels)
        valid_count = loss_mask.sum().clamp_min(1)
        return (losses * loss_mask).sum() / valid_count, valid_count

    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        loss, _ = self.forward_with_count(logits, input_ids, segment_ids)
        return loss
