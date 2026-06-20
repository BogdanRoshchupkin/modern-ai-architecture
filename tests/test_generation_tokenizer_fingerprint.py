import pytest
import torch

from src.tokenization.bpe import BpeTokenizer, bpe_file_sha256
from src.training.generation import validate_checkpoint_tokenizer


def _save_tokenizer(path, texts):
    tokenizer = BpeTokenizer(vocab_size=40)
    tokenizer.train(texts)
    tokenizer.save(path)
    return tokenizer


def test_validate_checkpoint_tokenizer_accepts_matching_fingerprint(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    _save_tokenizer(tokenizer_path, ["alpha beta", "alpha gamma"])
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(
        {"hyper_parameters": {"paths": {"tokenizer_sha256": bpe_file_sha256(tokenizer_path)}}},
        checkpoint_path,
    )

    validate_checkpoint_tokenizer(str(checkpoint_path), str(tokenizer_path))


def test_validate_checkpoint_tokenizer_rejects_mismatched_fingerprint(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    other_tokenizer_path = tmp_path / "other_tokenizer.json"
    _save_tokenizer(tokenizer_path, ["alpha beta", "alpha gamma"])
    _save_tokenizer(other_tokenizer_path, ["zeta omega", "theta omega"])
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(
        {"hyper_parameters": {"paths": {"tokenizer_sha256": bpe_file_sha256(tokenizer_path)}}},
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="Tokenizer fingerprint does not match"):
        validate_checkpoint_tokenizer(str(checkpoint_path), str(other_tokenizer_path))
