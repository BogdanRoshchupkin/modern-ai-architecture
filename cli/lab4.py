from __future__ import annotations

import argparse

from cli.lab2 import cmd_train, load_config
from src.training.generation import generate_text


def cmd_generate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    use_kv_cache = bool(config.generation.get("use_kv_cache", True)) and not args.no_kv_cache
    text = generate_text(
        checkpoint_path=args.checkpoint,
        config=config,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer or str(config.paths.tokenizer),
        max_new_tokens=args.max_new_tokens or int(config.generation.max_new_tokens),
        temperature=args.temperature or float(config.generation.temperature),
        top_k=args.top_k if args.top_k is not None else int(config.generation.top_k),
        use_kv_cache=use_kv_cache,
    )
    print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab 4 GQA training and KV-cache inference CLI")
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train")
    train.add_argument("--config", default="configs/lab4_gqa.yaml")
    train.add_argument("--data")
    train.add_argument("--resume-from-checkpoint")
    train.add_argument("--fast-dev-run", action="store_true")
    train.set_defaults(func=cmd_train)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", default="configs/lab4_gqa.yaml")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--prompt", default="The model")
    generate.add_argument("--tokenizer")
    generate.add_argument("--max-new-tokens", type=int)
    generate.add_argument("--temperature", type=float)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--no-kv-cache", action="store_true")
    generate.set_defaults(func=cmd_generate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
