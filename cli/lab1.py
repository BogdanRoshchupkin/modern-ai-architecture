from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.data.cleaning import clean_records
from src.data.download import download_file, resolve_common_crawl_warc_url
from src.data.packed_batching import pack_sequences
from src.data.warc import iter_warc_texts
from src.data.wikitext import iter_wikitext
from src.models.entropy import (
    Gpt2EntropyScorer,
    add_entropy_scores,
    dataset_information_density,
    deduplicate,
    filter_by_entropy_quantiles,
)
from src.tokenization.basic import CharTokenizer, WordTokenizer
from src.tokenization.bpe import BpeTokenizer
from src.utils.io import read_jsonl, write_jsonl


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def cmd_download_cc(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    url = args.url
    if url is None:
        url = resolve_common_crawl_warc_url(
            snapshot=args.snapshot or config["common_crawl_snapshot"],
            base_url=config["common_crawl_base_url"],
            warc_index=args.warc_index,
        )
        print(f"Resolved WARC URL: {url}")
    output = args.output or "data/raw/common_crawl.warc.gz"
    path = download_file(url, output)
    print(f"Downloaded WARC: {path}")


def cmd_warc_to_text(args: argparse.Namespace) -> None:
    rows = iter_warc_texts(args.input, limit=args.limit)
    count = write_jsonl(args.output, rows)
    print(f"Converted WARC records to text: {count} -> {args.output}")


def cmd_clean(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    rows = clean_records(
        read_jsonl(args.input),
        language_allowlist=args.languages or config["language_allowlist"],
        min_words=args.min_words,
        max_words=args.max_words or config["max_words_per_object"],
    )
    count = write_jsonl(args.output, rows)
    print(f"Cleaned records: {count} -> {args.output}")


def cmd_score_entropy(args: argparse.Namespace) -> None:
    scorer = Gpt2EntropyScorer(model_name=args.model, max_length=args.max_length)
    rows = list(add_entropy_scores(read_jsonl(args.input), scorer))
    density = dataset_information_density(rows)
    count = write_jsonl(args.output, rows)
    print(f"Scored records: {count} -> {args.output}")
    print(f"Dataset information density: {density:.4f} nats/token")


def cmd_filter_quality(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.input))
    unique = deduplicate(rows)
    filtered = filter_by_entropy_quantiles(unique, low=args.low_quantile, high=args.high_quantile)
    count = write_jsonl(args.output, filtered)
    duplicate_density = len(rows) / max(len(unique), 1)
    print(f"Input records: {len(rows)}")
    print(f"Unique records: {len(unique)}")
    print(f"Duplicate density: {duplicate_density:.4f}")
    print(f"Quality-filtered records: {count} -> {args.output}")


def cmd_tokenize(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.input))
    texts = [row["text"] for row in rows]
    if not texts:
        raise ValueError("No texts found for tokenization.")
    sample = random.choice(texts)

    char_tokenizer = CharTokenizer()
    char_tokenizer.train(texts)
    char_ids = char_tokenizer.encode(sample)
    print(f"Char tokenizer vocab size: {char_tokenizer.vocab_size}")
    print(f"Char tokenizer random object length: {len(char_ids)}")

    word_tokenizer = WordTokenizer(max_vocab_size=args.word_vocab_size)
    word_tokenizer.train(texts[: args.word_train_limit] if args.word_train_limit else texts)
    word_ids = word_tokenizer.encode(sample)
    print(f"Word tokenizer vocab size: {word_tokenizer.vocab_size}")
    print(f"Word tokenizer random object length: {len(word_ids)}")

    if args.bpe_load:
        bpe_tokenizer = BpeTokenizer.load(args.bpe_load)
        print(f"Loaded BPE tokenizer: {args.bpe_load}")
    else:
        bpe_tokenizer = BpeTokenizer(vocab_size=args.bpe_vocab_size)
        bpe_tokenizer.train(
            texts,
            max_documents=args.bpe_train_limit,
            min_pair_frequency=args.bpe_min_pair_frequency,
        )
        bpe_tokenizer.save(args.bpe_output)
        print(f"Saved BPE tokenizer: {args.bpe_output}")
    bpe_ids = bpe_tokenizer.encode(sample)
    print(f"BPE tokenizer vocab size: {bpe_tokenizer.vocab_size}")
    print(f"BPE tokenizer random object length: {len(bpe_ids)}")


def cmd_prepare_wikitext(args: argparse.Namespace) -> None:
    raw_count = write_jsonl(args.raw_output, iter_wikitext(split=args.split, name=args.name, limit=args.limit))
    print(f"Downloaded wikitext records: {raw_count} -> {args.raw_output}")
    config = load_config(args.config)
    cleaned = clean_records(
        read_jsonl(args.raw_output),
        language_allowlist=args.languages or config["language_allowlist"],
        min_words=args.min_words,
        max_words=args.max_words or config["max_words_per_object"],
    )
    clean_count = write_jsonl(args.clean_output, cleaned)
    print(f"Cleaned wikitext records: {clean_count} -> {args.clean_output}")


def cmd_pack(args: argparse.Namespace) -> None:
    tokenizer = BpeTokenizer.load(args.tokenizer)
    texts = [row["text"] for row in read_jsonl(args.input)]
    sequences = [tokenizer.encode(text) for text in texts]
    input_ids_tensor, attention_mask_tensor = pack_sequences(sequences, max_length=args.max_length, pad_id=args.pad_id)
    rows = [
        {"input_ids": input_ids, "attention_mask": attention_mask}
        for input_ids, attention_mask in zip(
            input_ids_tensor.tolist(),
            attention_mask_tensor.tolist(),
            strict=True,
        )
    ]
    count = write_jsonl(args.output, rows)
    print(f"Packed batches: {count} -> {args.output}")
    if rows:
        print(f"First packed sequence length: {len(rows[0]['input_ids'])}")
        print(f"First packed mask segment ids: {sorted(set(rows[0]['attention_mask']))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab 1 data preparation CLI")
    parser.set_defaults(func=None)
    parser.add_argument("--config", default="configs/default.json")
    subparsers = parser.add_subparsers(dest="command")

    download_cc = subparsers.add_parser("download-cc")
    download_cc.add_argument("--url")
    download_cc.add_argument("--snapshot")
    download_cc.add_argument("--warc-index", type=int, default=0)
    download_cc.add_argument("--output")
    download_cc.set_defaults(func=cmd_download_cc)

    warc_to_text = subparsers.add_parser("warc-to-text")
    warc_to_text.add_argument("--input", default="data/raw/common_crawl.warc.gz")
    warc_to_text.add_argument("--output", default="data/processed/common_crawl_text.jsonl")
    warc_to_text.add_argument("--limit", type=int)
    warc_to_text.set_defaults(func=cmd_warc_to_text)

    clean = subparsers.add_parser("clean")
    clean.add_argument("--input", default="data/processed/common_crawl_text.jsonl")
    clean.add_argument("--output", default="data/processed/common_crawl_clean.jsonl")
    clean.add_argument("--languages", nargs="+")
    clean.add_argument("--min-words", type=int, default=20)
    clean.add_argument("--max-words", type=int)
    clean.set_defaults(func=cmd_clean)

    score_entropy = subparsers.add_parser("score-entropy")
    score_entropy.add_argument("--input", default="data/processed/common_crawl_clean.jsonl")
    score_entropy.add_argument("--output", default="data/processed/common_crawl_entropy.jsonl")
    score_entropy.add_argument("--model", default="gpt2")
    score_entropy.add_argument("--max-length", type=int, default=1024)
    score_entropy.set_defaults(func=cmd_score_entropy)

    filter_quality = subparsers.add_parser("filter-quality")
    filter_quality.add_argument("--input", default="data/processed/common_crawl_entropy.jsonl")
    filter_quality.add_argument("--output", default="data/processed/common_crawl_quality.jsonl")
    filter_quality.add_argument("--low-quantile", type=float, default=0.02)
    filter_quality.add_argument("--high-quantile", type=float, default=0.98)
    filter_quality.set_defaults(func=cmd_filter_quality)

    tokenize = subparsers.add_parser("tokenize")
    tokenize.add_argument("--input", default="data/processed/common_crawl_quality.jsonl")
    tokenize.add_argument("--word-vocab-size", type=int)
    tokenize.add_argument("--word-train-limit", type=int)
    tokenize.add_argument("--bpe-vocab-size", type=int, default=8000)
    tokenize.add_argument("--bpe-output", default="data/processed/common_crawl_bpe.json")
    tokenize.add_argument("--bpe-load")
    tokenize.add_argument("--bpe-train-limit", type=int)
    tokenize.add_argument("--bpe-min-pair-frequency", type=int, default=2)
    tokenize.set_defaults(func=cmd_tokenize)

    prepare_wikitext = subparsers.add_parser("prepare-wikitext")
    prepare_wikitext.add_argument("--split", default="train")
    prepare_wikitext.add_argument("--name", default="wikitext-2-raw-v1")
    prepare_wikitext.add_argument("--limit", type=int)
    prepare_wikitext.add_argument("--raw-output", default="data/processed/wikitext_raw.jsonl")
    prepare_wikitext.add_argument("--clean-output", default="data/processed/wikitext_clean.jsonl")
    prepare_wikitext.add_argument("--languages", nargs="+")
    prepare_wikitext.add_argument("--min-words", type=int, default=20)
    prepare_wikitext.add_argument("--max-words", type=int)
    prepare_wikitext.set_defaults(func=cmd_prepare_wikitext)

    pack = subparsers.add_parser("pack")
    pack.add_argument("--input", default="data/processed/wikitext_quality.jsonl")
    pack.add_argument("--tokenizer", default="data/processed/common_crawl_bpe.json")
    pack.add_argument("--output", default="data/processed/wikitext_packed.jsonl")
    pack.add_argument("--max-length", type=int, default=512)
    pack.add_argument("--pad-id", type=int, default=0)
    pack.set_defaults(func=cmd_pack)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.func is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
