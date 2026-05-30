from src.tokenization.basic import CharTokenizer, WordTokenizer
from src.tokenization.bpe import BpeTokenizer


def test_char_tokenizer_reports_vocab_and_lengths():
    tokenizer = CharTokenizer()
    tokenizer.train(["aba"])

    assert tokenizer.vocab_size == 4
    assert len(tokenizer.encode("aba")) == 3


def test_word_tokenizer_splits_words_and_punctuation():
    tokenizer = WordTokenizer()
    tokenizer.train(["Hello, world!"])

    assert tokenizer.vocab_size == 6
    assert len(tokenizer.encode("Hello, world!")) == 4


def test_custom_bpe_tokenizer_trains_encodes_and_decodes():
    tokenizer = BpeTokenizer(vocab_size=30)
    tokenizer.train(["low lower newest widest", "low lowest newer"])

    ids = tokenizer.encode("low newer")
    decoded = tokenizer.decode(ids)

    assert tokenizer.vocab_size <= 30
    assert ids
    assert "low" in decoded
    assert "newer" in decoded


def test_custom_bpe_tokenizer_saves_and_loads(tmp_path):
    tokenizer = BpeTokenizer(vocab_size=30)
    tokenizer.train(["alpha beta alpha", "beta gamma"])
    path = tmp_path / "bpe.json"

    tokenizer.save(path)
    loaded = BpeTokenizer.load(path)

    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode("alpha beta") == tokenizer.encode("alpha beta")
