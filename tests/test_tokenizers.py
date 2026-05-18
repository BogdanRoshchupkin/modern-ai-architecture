from src.tokenization.basic import CharTokenizer, WordTokenizer


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
